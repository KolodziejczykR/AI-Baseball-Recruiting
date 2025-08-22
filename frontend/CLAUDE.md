# CLAUDE.md - Frontend Development Guide

This file provides guidance to Claude Code for frontend development of the AI Baseball Recruitment Platform.

## Project Overview

This is the frontend for **BaseballPATH**, an AI/ML-assisted baseball recruitment platform that predicts college level classifications (Non D1, Non P4 D1, Power 4 D1) for baseball players based on their statistics and position. The platform currently supports infielders, outfielders, and catchers, with pitcher models coming soon.

**Current Status**: Waitlist phase with full Supabase integration for data collection and user research.

## Brand Identity & Styling

### Company/Product Name
**BaseballPATH** (baseballpath.com domain to be purchased)

### Color Scheme & Visual Design
- **Primary Gradient**: Blue to red (`from-blue-600 to-red-600`) for main branding
- **Secondary Gradients**: Blue to rose (`from-blue-600 to-rose-600`) for headlines
- **Accent Colors**: Green to teal gradients (`from-emerald-500 to-teal-500`) for giveaway/value props
- **Background Theme**: 
  - **All Pages**: Clean slate blue gradient (`from-slate-50 via-blue-50 to-slate-100`) - consistent across waitlist, survey, and success pages

### Typography & Tone
- **Fonts**: Geist Sans & Geist Mono (modern, clean)
- **Tone**: Mix of professional/data-driven and friendly startup approach
- **Key Phrases**: "Trusted by coaches. Built by players. Powered by AI."
- **Style**: Modern, next-generation web design (inspired by Cursor.com aesthetic)

## Current Implementation: Waitlist Funnel

### Page Structure (Fully Implemented)
1. **`/waitlist`** - Main landing page with email collection
2. **`/waitlist/survey`** - Multi-question survey for market research  
3. **`/waitlist/success`** - Final confirmation page

### Waitlist Landing Page (`/waitlist`)
**Visual Hierarchy Priority**: Giveaway → Join Waitlist → Product Details

**Content Structure**:
- **Header**: BaseballPATH branding with blue-to-red gradient
- **Hero Headline**: "The AI That Gets You Recruited" 
- **Subheading**: "Trusted by coaches. Built by players. Powered by AI. After interviewing coaches across the country and living the recruiting struggle ourselves, we built the solution."
- **Giveaway CTA**: "Join the waitlist below and get entered into our launch giveaway! Five lucky waitlist members get BaseballPATH completely free on launch day." (with green-to-orange gradient)
- **Email Form**: Single input with "Reserve Your Spot" button
- **Legal Text**: "No purchase necessary • Winners drawn at launch • 100% free to enter"
- **3 Benefit Cards**: 
  1. "Supercharge your recruitment" (lightning bolt icon)
  2. "Shift your focus to development" (checkmark icon)  
  3. "Don't miss the giveaway" (clock icon)

**Background**: Clean slate-blue gradient (`from-slate-50 via-blue-50 to-slate-100`) for professional, minimalist appearance

### Survey Page (`/waitlist/survey`)
**Purpose**: Market research and user segmentation

**Question Flow**:
1. **Budget** (Required, 4-quadrant Kahoot-style): "$99 or below", "$99 to $199", "$199 to $399", "Above $399"
2. **Travel Team** (Required, 2-half): "Do you currently play for a summer baseball travel team?"
3. **Recruiting Agency** (Required, 2-half): "Do you already use a recruiting agency?" 
4. **Graduation Year** (Required, dropdown): 2026, 2027, 2028, 2029
5. **Recruiting Challenge** (Optional, textarea): "What's your biggest recruiting challenge or pain point?"
6. **Desired Features** (Optional, textarea): "What features would you love to see alongside the school matching algorithm?"
7. **Additional Info** (Optional, textarea): "Is there anything else you'd like for the team to know?"

**UI Design**:
- **Multiple Choice**: Kahoot-style buttons with blue-to-red gradient backgrounds (`from-blue-50 to-red-50`)
- **Selection States**: Blue ring and shadow when selected
- **Hover Effects**: Scale and shadow transitions
- **Required Fields**: Red asterisks (*) for visual indication
- **Form Validation**: Prevents submission until all required fields completed

**Background**: Same clean slate-blue gradient (`from-slate-50 via-blue-50 to-slate-100`) for visual consistency

### Success Page (`/waitlist/success`)
**Content**: Final confirmation with green checkmark icon, "Perfect! You're all set!" message, and follow-up instructions.

**Background**: Consistent slate-blue gradient (`from-slate-50 via-blue-50 to-slate-100`) with centered white card design

## Supabase Database Integration

### Database Schema
**Table**: `waitlist` (single table approach - all data in one row)

```sql
CREATE TABLE waitlist (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  email TEXT NOT NULL UNIQUE,
  budget TEXT,
  travel_team TEXT,
  recruiting_agency TEXT,
  graduation_year TEXT,
  recruiting_challenge TEXT,
  desired_features TEXT,
  additional_info TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### Data Flow
1. **Email Submission**: Creates initial record with just email
2. **Survey Completion**: Updates same record with all survey responses via `.update()` query
3. **Duplicate Prevention**: Checks email exists before allowing signup
4. **Error Handling**: User-friendly messages for database errors

### Environment Configuration
```bash
NEXT_PUBLIC_SUPABASE_URL=https://egzvnzykjpmaxilygnso.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=[supabase_anon_key]
```

### Security Policies
- **Row Level Security**: Enabled with public INSERT, UPDATE, and SELECT policies
- **Email Uniqueness**: Enforced at database level
- **Data Validation**: TypeScript interfaces for type safety

## Technical Implementation

### Frontend Stack
- **Framework**: Next.js 15 with App Router
- **Language**: TypeScript with strict typing
- **Styling**: Tailwind CSS v4 with custom gradient utilities
- **Database**: Supabase with `@supabase/supabase-js` client
- **Components**: Custom UI library (`/src/components/ui/`)
- **State Management**: React useState with sessionStorage for flow continuity
- **Icons**: Heroicons React + custom SVG icons

### Key Components
- **`Button`**: Multiple variants (primary, secondary, outline) with loading states
- **`Input`**: Email validation with error display
- **`Supabase Client`**: Centralized database operations
- **`Utils`**: Email validation, database wrappers, utility functions

### Development Commands
```bash
# Start development server
npm run dev

# Build for production  
npm run build

# Start production server
npm start

# Run linting
npm run lint
```

### File Structure
```
frontend/src/
├── app/
│   ├── waitlist/
│   │   ├── page.tsx (main landing)
│   │   ├── survey/page.tsx (survey form)
│   │   └── success/page.tsx (confirmation)
│   ├── globals.css
│   └── layout.tsx
├── components/ui/
│   ├── button.tsx
│   └── input.tsx
├── lib/
│   ├── supabase.ts (database config)
│   └── utils.ts (helper functions)
└── types/ (for future type definitions)
```

## Backend API Integration (Future)

### Planned ML Prediction Endpoints
- `POST /infielder/predict` - Infielders (SS, 2B, 3B, 1B)
- `POST /outfielder/predict` - Outfielders (OF)  
- `POST /catcher/predict` - Catchers (C)

### Prediction System Architecture
1. **Stage 1**: D1 vs Non-D1 classification
2. **Stage 2**: If D1 predicted, classify Power 4 vs Non-P4 D1

### Required Data by Position
- **Common**: height, weight, exit_velo_max, sixty_time, throwing_hand, hitting_handedness, player_region
- **Infielders**: `inf_velo`
- **Outfielders**: `of_velo`  
- **Catchers**: `c_velo`, `pop_time`

## Future Development Roadmap

### Immediate Next Steps
1. **Domain Purchase**: baseballpath.com acquisition and deployment
2. **Email Collection**: Continue building waitlist with current funnel
3. **Analytics**: Implement tracking for conversion metrics

### Phase 2: Core Platform
1. **Prediction Interface**: Integrate ML models with form-based input
2. **User Accounts**: Authentication and saved predictions
3. **Results Display**: Visual charts and recommendations
4. **Pricing Implementation**: Subscription or one-time payment model

### Phase 3: Enhanced Features
1. **Pitcher Models**: Expand beyond hitters
2. **Coach Dashboard**: School-side interface for prospect discovery
3. **Roster Integration**: Automated player data imports
4. **LLM Features**: Personalized recruiting advice

## Design Patterns & Guidelines

### Visual Consistency
- **Gradients**: Use established blue-red, green-orange, and background patterns
- **Spacing**: Consistent padding and margin scales
- **Typography**: Maintain hierarchy with font weights and sizes
- **Interactive Elements**: Hover effects, loading states, and transitions

### User Experience Principles
1. **Progressive Disclosure**: Start simple (email) → detailed (survey) → confirmation
2. **Clear Value Proposition**: Emphasize free giveaway and AI benefits
3. **Minimal Friction**: Required fields only for core data, optional for insights
4. **Error Prevention**: Validation, duplicate checking, and clear messaging
5. **Mobile-First**: Responsive design for all screen sizes

### Development Best Practices
1. **Type Safety**: Use TypeScript interfaces for all data structures
2. **Error Handling**: Graceful degradation with user-friendly messages
3. **Performance**: Optimize for fast loading and minimal bundle size
4. **Accessibility**: Proper ARIA labels and keyboard navigation
5. **SEO**: Appropriate meta tags and semantic HTML structure

## Current Status & Metrics

### Completed Features ✅
- [x] Waitlist landing page with Carolina blue theme
- [x] Multi-step survey with Kahoot-style interactions  
- [x] Supabase database integration with single-table design
- [x] Form validation and error handling
- [x] Responsive design across all breakpoints
- [x] Giveaway messaging and legal compliance
- [x] End-to-end user flow testing

### Ready for Launch
The current implementation is production-ready for waitlist collection and market research. All major functionality is implemented, tested, and integrated with Supabase for real-time data collection.